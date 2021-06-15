function    [ st, msg ] = cmd_rmdir( folderspec )       
%   cmd_rmdir removes a directory and its contents 
%   
%   Removes all directories and files in the specified directory in
%   addition to the directory itself.  Used to remove a directory tree.
%   See also: xtests\generic_utilies_test.m
           
    narginchk( 1, 1 )
    
    dos_cmd = sprintf( 'rmdir /S /Q "%s"', folderspec );
    
    [ st, msg ] = system( dos_cmd );
end